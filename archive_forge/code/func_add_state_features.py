from collections import Counter
from threading import Timer
import logging
import inspect
from ..core import MachineError, listify, State
def add_state_features(*args):
    """ State feature decorator. Should be used in conjunction with a custom Machine class. """

    def _class_decorator(cls):

        class CustomState(type('CustomState', args, {}), cls.state_cls):
            """ The decorated State. It is based on the State class used by the decorated Machine. """
        method_list = sum([c.dynamic_methods for c in inspect.getmro(CustomState) if hasattr(c, 'dynamic_methods')], [])
        CustomState.dynamic_methods = list(set(method_list))
        cls.state_cls = CustomState
        return cls
    return _class_decorator