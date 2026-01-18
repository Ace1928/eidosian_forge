import functools
import warnings
import json
import contextvars
import flask
from . import exceptions
from ._utils import AttributeDict
@staticmethod
@has_context
def record_timing(name, duration=None, description=None):
    """Records timing information for a server resource.

        :param name: The name of the resource.
        :type name: string

        :param duration: The time in seconds to report. Internally, this
            is rounded to the nearest millisecond.
        :type duration: float or None

        :param description: A description of the resource.
        :type description: string or None
        """
    timing_information = getattr(flask.g, 'timing_information', {})
    if name in timing_information:
        raise KeyError(f'Duplicate resource name "{name}" found.')
    timing_information[name] = {'dur': round(duration * 1000), 'desc': description}
    setattr(flask.g, 'timing_information', timing_information)