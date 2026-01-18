import collections
import logging
import pytest
import modin.logging
from modin.config import LogMode
def test_function_decorator_on_outer_function_6237(monkeypatch, get_log_messages):

    @modin.logging.enable_logging
    def inner_func():
        raise ValueError()

    @modin.logging.enable_logging
    def outer_func():
        inner_func()
    with monkeypatch.context() as ctx:
        mock_get_logger(ctx)
        with pytest.raises(ValueError):
            outer_func()
    assert get_log_messages('modin.logger.errors')['exception'] == ['STOP::PANDAS-API::inner_func']