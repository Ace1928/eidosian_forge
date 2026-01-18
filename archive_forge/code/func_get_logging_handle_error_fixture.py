import fixtures
def get_logging_handle_error_fixture():
    """returns a fixture to make logging raise formatting exceptions.

    To use::

      from oslo_log import fixture as log_fixture

      self.useFixture(log_fixture.get_logging_handle_error_fixture())
    """
    return fixtures.MonkeyPatch('logging.Handler.handleError', _handleError)