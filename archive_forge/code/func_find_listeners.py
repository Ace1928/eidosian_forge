from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
from thinc.api import Config, Model, Optimizer, set_dropout_rate
from ..errors import Errors
from ..language import Language
from ..tokens import Doc
from ..training import Example, validate_examples, validate_get_examples
from ..vocab import Vocab
from .trainable_pipe import TrainablePipe
def find_listeners(self, component) -> None:
    """Walk over a model of a processing component, looking for layers that
        are Tok2vecListener subclasses that have an upstream_name that matches
        this component. Listeners can also set their upstream_name attribute to
        the wildcard string '*' to match any `Tok2Vec`.

        You're unlikely to ever need multiple `Tok2Vec` components, so it's
        fine to leave your listeners upstream_name on '*'.
        """
    names = ('*', self.name)
    if isinstance(getattr(component, 'model', None), Model):
        for node in component.model.walk():
            if isinstance(node, Tok2VecListener) and node.upstream_name in names:
                self.add_listener(node, component.name)