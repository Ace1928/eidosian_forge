from parlai.core.build_data import modelzoo_path
from parlai.core.loader import load_agent_module
from parlai.core.loader import register_agent  # noqa: F401
from parlai.core.opt import Opt
from parlai.utils.misc import warn_once
import copy
import os
import parlai.utils.logging as logging
def create_agent_from_model_file(model_file, opt_overrides=None):
    """
    Load agent from model file if it exists.

    :param opt_overrides:
        An optional dict of option overrides can also be provided.
    :return:
        The agent
    """
    opt = {}
    add_datapath_and_model_args(opt)
    opt['model_file'] = modelzoo_path(opt.get('datapath'), model_file)
    if opt_overrides is None:
        opt_overrides = {}
    opt['override'] = opt_overrides
    return create_agent_from_opt_file(opt)