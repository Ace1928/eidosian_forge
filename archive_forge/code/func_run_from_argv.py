import sys
from django.conf import settings
from django.core.management.base import BaseCommand
from django.core.management.utils import get_command_line_option
from django.test.runner import get_max_test_processes
from django.test.utils import NullTimeKeeper, TimeKeeper, get_runner
def run_from_argv(self, argv):
    """
        Pre-parse the command line to extract the value of the --testrunner
        option. This allows a test runner to define additional command line
        arguments.
        """
    self.test_runner = get_command_line_option(argv, '--testrunner')
    super().run_from_argv(argv)