import signal
from django.db.backends.base.client import BaseDatabaseClient
def runshell(self, parameters):
    sigint_handler = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        super().runshell(parameters)
    finally:
        signal.signal(signal.SIGINT, sigint_handler)