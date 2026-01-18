import matplotlib.pyplot as plt
import copy
import pprint
import os.path as path
from   mplfinance._arg_validators import _process_kwargs, _validate_vkwargs_dict
from   mplfinance._styledata      import _styles
from   mplfinance._helpers        import _mpf_is_color_like
def make_mpf_style(**kwargs):
    config = _process_kwargs(kwargs, _valid_make_mpf_style_kwargs())
    if config['rc'] is not None and config['legacy_rc'] is not None:
        raise ValueError('kwargs `rc` and `legacy_rc` may NOT be used together!')
    if config['base_mpf_style'] is not None:
        style = _get_mpfstyle(config['base_mpf_style'])
        if config['rc'] is not None:
            rc = config['rc']
            del config['rc']
            if isinstance(style['rc'], list):
                style['rc'] = dict(style['rc'])
            if style['rc'] is None:
                style['rc'] = {}
            style['rc'].update(rc)
        elif config['legacy_rc'] is not None:
            config['rc'] = config['legacy_rc']
            del config['legacy_rc']
        update = [(k, v) for k, v in config.items() if v is not None]
        style.update(update)
    else:
        style = config
    if style['marketcolors'] is None:
        style['marketcolors'] = _styles['default']['marketcolors']
    return style