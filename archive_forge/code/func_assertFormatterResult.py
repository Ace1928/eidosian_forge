import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def assertFormatterResult(self, formatter, who, result):
    formatter_kwargs = dict()
    if who is not None:
        author_list_handler = log.author_list_registry.get(who)
        formatter_kwargs['author_list_handler'] = author_list_handler
    TestCaseForLogFormatter.assertFormatterResult(self, result, self.wt.branch, formatter, formatter_kwargs=formatter_kwargs)