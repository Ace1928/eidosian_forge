from typing import Optional, List
from thinc.types import Floats2d
from thinc.api import Model, zero_init, use_ops
from spacy.tokens import Doc
from spacy.compat import Literal
from spacy.errors import Errors
from spacy.util import registry
def resize_output_v1(model, new_nO):
    Linear = registry.get('layers', 'Linear.v1')
    lower = model.get_ref('lower')
    upper = model.get_ref('upper')
    if not model.attrs['has_upper']:
        if lower.has_dim('nO') is None:
            lower.set_dim('nO', new_nO)
        return
    elif upper.has_dim('nO') is None:
        upper.set_dim('nO', new_nO)
        return
    elif new_nO == upper.get_dim('nO'):
        return
    smaller = upper
    nI = None
    if smaller.has_dim('nI'):
        nI = smaller.get_dim('nI')
    with use_ops('numpy'):
        larger = Linear(nO=new_nO, nI=nI)
        larger.init = smaller.init
    if nI:
        larger_W = larger.ops.alloc2f(new_nO, nI)
        larger_b = larger.ops.alloc1f(new_nO)
        smaller_W = smaller.get_param('W')
        smaller_b = smaller.get_param('b')
        if smaller.has_dim('nO'):
            larger_W[:smaller.get_dim('nO')] = smaller_W
            larger_b[:smaller.get_dim('nO')] = smaller_b
            for i in range(smaller.get_dim('nO'), new_nO):
                model.attrs['unseen_classes'].add(i)
        larger.set_param('W', larger_W)
        larger.set_param('b', larger_b)
    model._layers[-1] = larger
    model.set_ref('upper', larger)
    return model