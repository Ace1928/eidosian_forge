from __future__ import print_function, absolute_import
from textwrap import dedent
from shibokensupport.signature import inspect, typing
from shibokensupport.signature.mapping import ellipsis
from shibokensupport.signature.lib.tool import SimpleNamespace
def make_signature_nameless(signature):
    """
    Make a Signature Nameless

    We use an existing signature and change the type of its parameters.
    The signature looks different, but is totally intact.
    """
    for key in signature.parameters.keys():
        signature.parameters[key].__class__ = NamelessParameter