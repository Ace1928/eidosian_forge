import os
import re
import subprocess
import sys
import unittest
from io import BytesIO
from io import UnsupportedOperation as _UnsupportedOperation
import iso8601
from testtools import ExtendedToOriginalDecorator, content, content_type
from testtools.compat import _b, _u
from testtools.content import TracebackContent
from testtools import CopyStreamResult, testresult
from subunit import chunked, details
from subunit.v2 import ByteStreamToStreamResult, StreamResultToBytes
def tag_stream(original, filtered, tags):
    """Alter tags on a stream.

    :param original: The input stream.
    :param filtered: The output stream.
    :param tags: The tags to apply. As in a normal stream - a list of 'TAG' or
        '-TAG' commands.

        A 'TAG' command will add the tag to the output stream,
        and override any existing '-TAG' command in that stream.
        Specifically:
         * A global 'tags: TAG' will be added to the start of the stream.
         * Any tags commands with -TAG will have the -TAG removed.

        A '-TAG' command will remove the TAG command from the stream.
        Specifically:
         * A 'tags: -TAG' command will be added to the start of the stream.
         * Any 'tags: TAG' command will have 'TAG' removed from it.
        Additionally, any redundant tagging commands (adding a tag globally
        present, or removing a tag globally removed) are stripped as a
        by-product of the filtering.
    :return: 0
    """
    new_tags, gone_tags = tags_to_new_gone(tags)
    source = ByteStreamToStreamResult(original, non_subunit_name='stdout')

    class Tagger(CopyStreamResult):

        def status(self, **kwargs):
            tags = kwargs.get('test_tags')
            if not tags:
                tags = set()
            tags.update(new_tags)
            tags.difference_update(gone_tags)
            if tags:
                kwargs['test_tags'] = tags
            else:
                kwargs['test_tags'] = None
            super(Tagger, self).status(**kwargs)
    output = Tagger([StreamResultToBytes(filtered)])
    source.run(output)
    return 0