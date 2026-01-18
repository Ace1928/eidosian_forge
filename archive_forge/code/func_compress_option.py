import argparse
import os
from time import time
from pyzstd import compress_stream, decompress_stream, \
def compress_option(args):
    if args.threads == 0:
        threads_msg = 'single-thread mode'
    else:
        threads_msg = 'multi-thread mode, %d threads.' % args.threads
    if args.long >= 0:
        use_long = 1
        windowLog = args.long
        long_msg = 'yes, windowLog is %d.' % windowLog
    else:
        use_long = 0
        windowLog = 0
        long_msg = 'no'
    option = {CParameter.compressionLevel: args.level, CParameter.nbWorkers: args.threads, CParameter.enableLongDistanceMatching: use_long, CParameter.windowLog: windowLog, CParameter.checksumFlag: args.checksum, CParameter.dictIDFlag: args.write_dictID}
    msg = ' - compression level: {}\n - threads: {}\n - long mode: {}\n - zstd dictionary: {}\n - add checksum: {}'.format(args.level, threads_msg, long_msg, args.zd, args.checksum)
    print(msg)
    return option