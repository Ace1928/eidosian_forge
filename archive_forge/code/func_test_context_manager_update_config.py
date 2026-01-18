import os
import unittest.mock
import warnings
import pytest
from packaging import version
import modin.config as cfg
from modin.config.envvars import _check_vars
from modin.config.pubsub import _UNSET, ExactStr
@pytest.mark.parametrize('modify_config', [{cfg.RangePartitioning: False, cfg.LazyExecution: 'Auto'}], indirect=True)
def test_context_manager_update_config(modify_config):
    assert cfg.RangePartitioning.get() is False
    with cfg.context(RangePartitioning=True):
        assert cfg.RangePartitioning.get() is True
    assert cfg.RangePartitioning.get() is False
    assert cfg.RangePartitioning.get() is False
    with cfg.context(RangePartitioning=True):
        assert cfg.RangePartitioning.get() is True
        with cfg.context(RangePartitioning=False):
            assert cfg.RangePartitioning.get() is False
            with cfg.context(RangePartitioning=False):
                assert cfg.RangePartitioning.get() is False
            assert cfg.RangePartitioning.get() is False
        assert cfg.RangePartitioning.get() is True
    assert cfg.RangePartitioning.get() is False
    assert cfg.RangePartitioning.get() is False
    assert cfg.LazyExecution.get() == 'Auto'
    with cfg.context(RangePartitioning=True, LazyExecution='Off'):
        assert cfg.RangePartitioning.get() is True
        assert cfg.LazyExecution.get() == 'Off'
    assert cfg.RangePartitioning.get() is False
    assert cfg.LazyExecution.get() == 'Auto'
    assert cfg.RangePartitioning.get() is False
    assert cfg.LazyExecution.get() == 'Auto'
    with cfg.context(RangePartitioning=True, LazyExecution='Off'):
        assert cfg.RangePartitioning.get() is True
        assert cfg.LazyExecution.get() == 'Off'
        with cfg.context(RangePartitioning=False):
            assert cfg.RangePartitioning.get() is False
            assert cfg.LazyExecution.get() == 'Off'
            with cfg.context(LazyExecution='On'):
                assert cfg.RangePartitioning.get() is False
                assert cfg.LazyExecution.get() == 'On'
                with cfg.context(RangePartitioning=True, LazyExecution='Off'):
                    assert cfg.RangePartitioning.get() is True
                    assert cfg.LazyExecution.get() == 'Off'
                assert cfg.RangePartitioning.get() is False
                assert cfg.LazyExecution.get() == 'On'
            assert cfg.RangePartitioning.get() is False
            assert cfg.LazyExecution.get() == 'Off'
        assert cfg.RangePartitioning.get() is True
        assert cfg.LazyExecution.get() == 'Off'
    assert cfg.RangePartitioning.get() is False
    assert cfg.LazyExecution.get() == 'Auto'