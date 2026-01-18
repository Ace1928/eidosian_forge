import collections
import logging
import pytest
import modin.logging
from modin.config import LogMode
def test_function_decorator(monkeypatch, get_log_messages):

    @modin.logging.enable_logging
    def func(do_raise):
        if do_raise:
            raise ValueError()
    with monkeypatch.context() as ctx:
        mock_get_logger(ctx)
        func(do_raise=False)
        with pytest.raises(ValueError):
            func(do_raise=True)
    assert 'func' in get_log_messages()[logging.INFO][0]
    assert 'START' in get_log_messages()[logging.INFO][0]
    assert get_log_messages('modin.logger.errors')['exception'] == ['STOP::PANDAS-API::func']