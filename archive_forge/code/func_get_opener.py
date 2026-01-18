from __future__ import absolute_import, print_function, unicode_literals
import typing
import collections
import contextlib
import pkg_resources
from ..errors import ResourceReadOnly
from .base import Opener
from .errors import EntryPointError, UnsupportedProtocol
from .parse import parse_fs_url
def get_opener(self, protocol):
    """Get the opener class associated to a given protocol.

        Arguments:
            protocol (str): A filesystem protocol.

        Returns:
            Opener: an opener instance.

        Raises:
            ~fs.opener.errors.UnsupportedProtocol: If no opener
                could be found for the given protocol.
            EntryPointLoadingError: If the returned entry point
                is not an `Opener` subclass or could not be loaded
                successfully.

        """
    protocol = protocol or self.default_opener
    if self.load_extern:
        entry_point = next(pkg_resources.iter_entry_points('fs.opener', protocol), None)
    else:
        entry_point = None
    if entry_point is None:
        if protocol in self._protocols:
            opener_instance = self._protocols[protocol]
        else:
            raise UnsupportedProtocol("protocol '{}' is not supported".format(protocol))
    else:
        try:
            opener = entry_point.load()
        except Exception as exception:
            raise EntryPointError('could not load entry point; {}'.format(exception))
        if not issubclass(opener, Opener):
            raise EntryPointError('entry point did not return an opener')
        try:
            opener_instance = opener()
        except Exception as exception:
            raise EntryPointError('could not instantiate opener; {}'.format(exception))
    return opener_instance