import logging as std_logging
import os
import fixtures
def delete_stored_logs(self):
    self.logger._output.truncate(0)