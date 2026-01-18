from __future__ import annotations
from babel.messages import frontend
def check_message_extractors(dist, name, value):
    """Validate the ``message_extractors`` keyword argument to ``setup()``.

    :param dist: the distutils/setuptools ``Distribution`` object
    :param name: the name of the keyword argument (should always be
                 "message_extractors")
    :param value: the value of the keyword argument
    :raise `DistutilsSetupError`: if the value is not valid
    """
    assert name == 'message_extractors'
    if not isinstance(value, dict):
        raise SetupError('the value of the "message_extractors" parameter must be a dictionary')