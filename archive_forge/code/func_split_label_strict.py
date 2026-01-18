from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
@staticmethod
def split_label_strict(label):
    """
        Splits a label created with L{parse_label}.

        To parse labels with a less strict syntax, use the L{split_label_fuzzy}
        method instead.

        @warning: This method only parses the label, it doesn't make sure the
            label actually points to a valid memory location.

        @type  label: str
        @param label: Label to split.

        @rtype:  tuple( str or None, str or int or None, int or None )
        @return: Tuple containing the C{module} name,
            the C{function} name or ordinal, and the C{offset} value.

            If the label doesn't specify a module,
            then C{module} is C{None}.

            If the label doesn't specify a function,
            then C{function} is C{None}.

            If the label doesn't specify an offset,
            then C{offset} is C{0}.

        @raise ValueError: The label is malformed.
        """
    module = function = None
    offset = 0
    if not label:
        label = '0x0'
    else:
        label = label.replace(' ', '')
        label = label.replace('\t', '')
        label = label.replace('\r', '')
        label = label.replace('\n', '')
        if not label:
            label = '0x0'
    if '!' in label:
        try:
            module, function = label.split('!')
        except ValueError:
            raise ValueError('Malformed label: %s' % label)
        if function:
            if '+' in module:
                raise ValueError('Malformed label: %s' % label)
            if '+' in function:
                try:
                    function, offset = function.split('+')
                except ValueError:
                    raise ValueError('Malformed label: %s' % label)
                try:
                    offset = HexInput.integer(offset)
                except ValueError:
                    raise ValueError('Malformed label: %s' % label)
            else:
                try:
                    offset = HexInput.integer(function)
                    function = None
                except ValueError:
                    pass
        elif '+' in module:
            try:
                module, offset = module.split('+')
            except ValueError:
                raise ValueError('Malformed label: %s' % label)
            try:
                offset = HexInput.integer(offset)
            except ValueError:
                raise ValueError('Malformed label: %s' % label)
        else:
            try:
                offset = HexInput.integer(module)
                module = None
            except ValueError:
                pass
        if not module:
            module = None
        if not function:
            function = None
    else:
        try:
            offset = HexInput.integer(label)
        except ValueError:
            if label.startswith('#'):
                function = label
                try:
                    HexInput.integer(function[1:])
                except ValueError:
                    raise ValueError('Ambiguous label: %s' % label)
            else:
                raise ValueError('Ambiguous label: %s' % label)
    if function and function.startswith('#'):
        try:
            function = HexInput.integer(function[1:])
        except ValueError:
            pass
    if not offset:
        offset = None
    return (module, function, offset)