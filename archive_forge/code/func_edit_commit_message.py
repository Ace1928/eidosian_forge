import codecs
import os
import sys
from io import BytesIO, StringIO
from subprocess import call
from . import bedding, cmdline, config, osutils, trace, transport, ui
from .errors import BzrError
from .hooks import Hooks
def edit_commit_message(infotext, ignoreline=DEFAULT_IGNORE_LINE, start_message=None):
    """Let the user edit a commit message in a temp file.

    This is run if they don't give a message or
    message-containing file on the command line.

    :param infotext:    Text to be displayed at bottom of message
                        for the user's reference;
                        currently similar to 'bzr status'.

    :param ignoreline:  The separator to use above the infotext.

    :param start_message:   The text to place above the separator, if any.
                            This will not be removed from the message
                            after the user has edited it.

    :return:    commit message or None.
    """
    if start_message is not None:
        start_message = start_message.encode(osutils.get_user_encoding())
    infotext = infotext.encode(osutils.get_user_encoding(), 'replace')
    return edit_commit_message_encoded(infotext, ignoreline, start_message)