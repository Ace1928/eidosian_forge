from IPython.utils.capture import capture_output
import pytest
def test_alias_args_commented_nargs():
    """Check that alias correctly counts args, excluding those commented out"""
    am = _ip.alias_manager
    alias_name = 'comargcount'
    cmd = 'echo this is %%s a commented out arg and this is not %s'
    am.define_alias(alias_name, cmd)
    assert am.is_alias(alias_name)
    thealias = am.get_alias(alias_name)
    assert thealias.nargs == 1