import io
import logging
import pytest
from traitlets import default
from .mockextension import MockExtensionApp
from notebook_shim import shim
def list_test_params(param_input):
    """"""
    params = []
    for test in param_input:
        name, value = (test[0], test[1])
        option = '--MockExtensionApp.{name}={value}'.format(name=name, value=value)
        params.append([[option], name, value])
    return params