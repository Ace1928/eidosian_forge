import datetime
import sys
from functools import partial
from optparse import OptionGroup, OptionParser, OptionValueError
from subunit import make_stream_binary
from iso8601 import UTC
from subunit.v2 import StreamResultToBytes
def generate_stream_results(args, output_writer):
    output_writer.startTestRun()
    if args.attach_file:
        reader = partial(args.attach_file.read, _CHUNK_SIZE)
        this_file_hunk = reader()
        next_file_hunk = reader()
    is_first_packet = True
    is_last_packet = False
    while not is_last_packet:
        write_status = output_writer.status
        if is_first_packet:
            if args.attach_file:
                if args.mimetype:
                    write_status = partial(write_status, mime_type=args.mimetype)
            if args.tags:
                write_status = partial(write_status, test_tags=set(args.tags))
            write_status = partial(write_status, timestamp=create_timestamp())
            if args.action not in _FINAL_ACTIONS:
                write_status = partial(write_status, test_status=args.action)
            is_first_packet = False
        if args.attach_file:
            filename = args.file_name or args.attach_file.name
            write_status = partial(write_status, file_name=filename, file_bytes=this_file_hunk)
            if next_file_hunk == b'':
                write_status = partial(write_status, eof=True)
                is_last_packet = True
            else:
                this_file_hunk = next_file_hunk
                next_file_hunk = reader()
        else:
            is_last_packet = True
        if args.test_id:
            write_status = partial(write_status, test_id=args.test_id)
        if is_last_packet:
            if args.action in _FINAL_ACTIONS:
                write_status = partial(write_status, test_status=args.action)
        write_status()
    output_writer.stopTestRun()