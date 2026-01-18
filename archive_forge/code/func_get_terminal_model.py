from __future__ import annotations
import typing as t
from datetime import timedelta
from jupyter_server._tz import isoformat, utcnow
from jupyter_server.prometheus import metrics
from terminado.management import NamedTermManager, PtyWithClients
from tornado import web
from tornado.ioloop import IOLoop, PeriodicCallback
from traitlets import Integer
from traitlets.config import LoggingConfigurable
def get_terminal_model(self, name: str) -> MODEL:
    """Return a JSON-safe dict representing a terminal.
        For use in representing terminals in the JSON APIs.
        """
    self._check_terminal(name)
    term = self.terminals[name]
    return {'name': name, 'last_activity': isoformat(term.last_activity)}