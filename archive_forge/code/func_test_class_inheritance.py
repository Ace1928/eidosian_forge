import collections
import logging
import pytest
import modin.logging
from modin.config import LogMode
def test_class_inheritance(monkeypatch, get_log_messages):

    class Foo(modin.logging.ClassLogger, modin_layer='CUSTOM'):

        def method1(self):
            pass

    class Bar(Foo):

        def method2(self):
            pass
    with monkeypatch.context() as ctx:
        mock_get_logger(ctx)
        Foo().method1()
        Bar().method1()
        Bar().method2()
    assert get_log_messages()[logging.INFO] == ['START::CUSTOM::Foo.method1', 'STOP::CUSTOM::Foo.method1', 'START::CUSTOM::Foo.method1', 'STOP::CUSTOM::Foo.method1', 'START::CUSTOM::Bar.method2', 'STOP::CUSTOM::Bar.method2']