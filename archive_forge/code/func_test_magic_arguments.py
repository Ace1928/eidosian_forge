import argparse
import sys
from IPython.core.magic_arguments import (argument, argument_group, kwds,
def test_magic_arguments():
    options = 'optional arguments' if sys.version_info < (3, 10) else 'options'
    assert magic_foo1.__doc__ == f'::\n\n  %foo1 [-f FOO]\n\n{LEADING_SPACE}A docstring.\n\n{options}:\n  -f FOO, --foo FOO  an argument\n'
    assert getattr(magic_foo1, 'argcmd_name', None) == None
    assert real_name(magic_foo1) == 'foo1'
    assert magic_foo1(None, '') == argparse.Namespace(foo=None)
    assert hasattr(magic_foo1, 'has_arguments')
    assert magic_foo2.__doc__ == f'::\n\n  %foo2\n\n{LEADING_SPACE}A docstring.\n'
    assert getattr(magic_foo2, 'argcmd_name', None) == None
    assert real_name(magic_foo2) == 'foo2'
    assert magic_foo2(None, '') == argparse.Namespace()
    assert hasattr(magic_foo2, 'has_arguments')
    assert magic_foo3.__doc__ == f'::\n\n  %foo3 [-f FOO] [-b BAR] [-z BAZ]\n\n{LEADING_SPACE}A docstring.\n\n{options}:\n  -f FOO, --foo FOO  an argument\n\nGroup:\n  -b BAR, --bar BAR  a grouped argument\n\nSecond Group:\n  -z BAZ, --baz BAZ  another grouped argument\n'
    assert getattr(magic_foo3, 'argcmd_name', None) == None
    assert real_name(magic_foo3) == 'foo3'
    assert magic_foo3(None, '') == argparse.Namespace(bar=None, baz=None, foo=None)
    assert hasattr(magic_foo3, 'has_arguments')
    assert magic_foo4.__doc__ == f'::\n\n  %foo4 [-f FOO]\n\n{LEADING_SPACE}A docstring.\n\n{options}:\n  -f FOO, --foo FOO  an argument\n'
    assert getattr(magic_foo4, 'argcmd_name', None) == None
    assert real_name(magic_foo4) == 'foo4'
    assert magic_foo4(None, '') == argparse.Namespace()
    assert hasattr(magic_foo4, 'has_arguments')
    assert magic_foo5.__doc__ == f'::\n\n  %frobnicate [-f FOO]\n\n{LEADING_SPACE}A docstring.\n\n{options}:\n  -f FOO, --foo FOO  an argument\n'
    assert getattr(magic_foo5, 'argcmd_name', None) == 'frobnicate'
    assert real_name(magic_foo5) == 'frobnicate'
    assert magic_foo5(None, '') == argparse.Namespace(foo=None)
    assert hasattr(magic_foo5, 'has_arguments')
    assert magic_magic_foo.__doc__ == f'::\n\n  %magic_foo [-f FOO]\n\n{LEADING_SPACE}A docstring.\n\n{options}:\n  -f FOO, --foo FOO  an argument\n'
    assert getattr(magic_magic_foo, 'argcmd_name', None) == None
    assert real_name(magic_magic_foo) == 'magic_foo'
    assert magic_magic_foo(None, '') == argparse.Namespace(foo=None)
    assert hasattr(magic_magic_foo, 'has_arguments')
    assert foo.__doc__ == f'::\n\n  %foo [-f FOO]\n\n{LEADING_SPACE}A docstring.\n\n{options}:\n  -f FOO, --foo FOO  an argument\n'
    assert getattr(foo, 'argcmd_name', None) == None
    assert real_name(foo) == 'foo'
    assert foo(None, '') == argparse.Namespace(foo=None)
    assert hasattr(foo, 'has_arguments')