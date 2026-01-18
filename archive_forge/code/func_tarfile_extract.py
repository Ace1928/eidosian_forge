import argparse
import os
from time import time
from pyzstd import compress_stream, decompress_stream, \
def tarfile_extract(args):
    if args.input is None:
        msg = 'need to specify input file using -d/--decompress option.'
        raise FileNotFoundError(msg)
    input_file_size = os.path.getsize(args.input.name)
    if not os.path.isdir(args.tar_output_dir):
        msg = 'Tar archive output dir invalid: ' + args.tar_output_dir
        raise NotADirectoryError(msg)
    ZstdTarFile = get_ZstdTarFile()
    option = {DParameter.windowLogMax: args.windowLogMax}
    msg = 'Extract tar archive:\n - input file: {}\n - output dir: {}\n - zstd dictionary: {}\nExtracting, please wait.'.format(args.input.name, args.tar_output_dir, args.zd)
    print(msg, flush=True)
    t1 = time()
    with ZstdTarFile(args.input, mode='r', zstd_dict=args.zd, level_or_option=option) as f:
        f.extractall(args.tar_output_dir)
        uncompressed_size = f.fileobj.tell()
    t2 = time()
    close_files(args)
    if uncompressed_size != 0:
        ratio = 100 * input_file_size / uncompressed_size
    else:
        ratio = 100.0
    msg = 'Extraction succeeded, {:.2f} seconds.\nInput {:,} bytes, output ~{:,} bytes, ratio {:.2f}%.\n'.format(t2 - t1, input_file_size, uncompressed_size, ratio)
    print(msg)