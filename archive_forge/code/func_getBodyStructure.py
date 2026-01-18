import binascii
import codecs
import copy
import email.utils
import functools
import re
import string
import tempfile
import time
import uuid
from base64 import decodebytes, encodebytes
from io import BytesIO
from itertools import chain
from typing import Any, List, cast
from zope.interface import implementer
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer, error, interfaces
from twisted.internet.defer import maybeDeferred
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, text
from twisted.python.compat import (
def getBodyStructure(msg, extended=False):
    """
    RFC 3501, 7.4.2, BODYSTRUCTURE::

      A parenthesized list that describes the [MIME-IMB] body structure of a
      message.  This is computed by the server by parsing the [MIME-IMB] header
      fields, defaulting various fields as necessary.

        For example, a simple text message of 48 lines and 2279 octets can have
        a body structure of: ("TEXT" "PLAIN" ("CHARSET" "US-ASCII") NIL NIL
        "7BIT" 2279 48)

    This is represented as::

        ["TEXT", "PLAIN", ["CHARSET", "US-ASCII"], None, None, "7BIT", 2279, 48]

    These basic fields are documented in the RFC as:

      1. body type

         A string giving the content media type name as defined in
         [MIME-IMB].

      2. body subtype

         A string giving the content subtype name as defined in
         [MIME-IMB].

      3. body parameter parenthesized list

         A parenthesized list of attribute/value pairs [e.g., ("foo"
         "bar" "baz" "rag") where "bar" is the value of "foo" and
         "rag" is the value of "baz"] as defined in [MIME-IMB].

      4. body id

         A string giving the content id as defined in [MIME-IMB].

      5. body description

         A string giving the content description as defined in
         [MIME-IMB].

      6. body encoding

         A string giving the content transfer encoding as defined in
         [MIME-IMB].

      7. body size

         A number giving the size of the body in octets.  Note that this size is
         the size in its transfer encoding and not the resulting size after any
         decoding.

    Put another way, the body structure is a list of seven elements.  The
    semantics of the elements of this list are:

       1. Byte string giving the major MIME type
       2. Byte string giving the minor MIME type
       3. A list giving the Content-Type parameters of the message
       4. A byte string giving the content identifier for the message part, or
          None if it has no content identifier.
       5. A byte string giving the content description for the message part, or
          None if it has no content description.
       6. A byte string giving the Content-Encoding of the message body
       7. An integer giving the number of octets in the message body

    The RFC goes on::

        Multiple parts are indicated by parenthesis nesting.  Instead of a body
        type as the first element of the parenthesized list, there is a sequence
        of one or more nested body structures.  The second element of the
        parenthesized list is the multipart subtype (mixed, digest, parallel,
        alternative, etc.).

        For example, a two part message consisting of a text and a
        BASE64-encoded text attachment can have a body structure of: (("TEXT"
        "PLAIN" ("CHARSET" "US-ASCII") NIL NIL "7BIT" 1152 23)("TEXT" "PLAIN"
        ("CHARSET" "US-ASCII" "NAME" "cc.diff")
        "<960723163407.20117h@cac.washington.edu>" "Compiler diff" "BASE64" 4554
        73) "MIXED")

    This is represented as::

        [["TEXT", "PLAIN", ["CHARSET", "US-ASCII"], None, None, "7BIT", 1152,
          23],
         ["TEXT", "PLAIN", ["CHARSET", "US-ASCII", "NAME", "cc.diff"],
          "<960723163407.20117h@cac.washington.edu>", "Compiler diff",
          "BASE64", 4554, 73],
         "MIXED"]

    In other words, a list of N + 1 elements, where N is the number of parts in
    the message.  The first N elements are structures as defined by the previous
    section.  The last element is the minor MIME subtype of the multipart
    message.

    Additionally, the RFC describes extension data::

        Extension data follows the multipart subtype.  Extension data is never
        returned with the BODY fetch, but can be returned with a BODYSTRUCTURE
        fetch.  Extension data, if present, MUST be in the defined order.

    The C{extended} flag controls whether extension data might be returned with
    the normal data.
    """
    return _getMessageStructure(msg).encode(extended)