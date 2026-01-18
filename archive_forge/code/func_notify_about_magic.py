import sys
from _pydev_bundle.pydev_console_utils import BaseInterpreterInterface
import traceback
from _pydev_bundle.pydev_ipython_console_011 import get_pydev_frontend
def notify_about_magic(self):
    if not self.notification_succeeded:
        self.notification_tries += 1
        if self.notification_tries > self.notification_max_tries:
            return
        completions = self.getCompletions('%', '%')
        magic_commands = [x[0] for x in completions]
        server = self.get_server()
        if server is not None:
            try:
                server.NotifyAboutMagic(magic_commands, self.interpreter.is_automagic())
                self.notification_succeeded = True
            except:
                self.notification_succeeded = False