from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_childRestriction(self) -> None:
    """
        L{Document.appendChild} raises L{ValueError} if the document already
        has a child.
        """
    document = microdom.Document()
    child = microdom.Node()
    another = microdom.Node()
    document.appendChild(child)
    self.assertRaises(ValueError, document.appendChild, another)