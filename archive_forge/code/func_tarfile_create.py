import argparse
import os
from time import time
from pyzstd import compress_stream, decompress_stream, \
def tarfile_create(args):
    args.tar_input_dir = args.tar_input_dir.rstrip(os.sep)
    if not os.path.isdir(args.tar_input_dir):
        msg = 'Tar archive input dir invalid: ' + args.tar_input_dir
        raise NotADirectoryError(msg)
    dirname, basename = os.path.split(args.tar_input_dir)
    if args.output is None:
        out_path = os.path.join(dirname, basename + '.tar.zst')
        open_output(args, out_path)
    ZstdTarFile = get_ZstdTarFile()
    msg = 'Archive tar file:\n - input directory: {}\n - output file: {}'.format(args.tar_input_dir, args.output.name)
    print(msg)
    option = compress_option(args)
    print('Archiving, please wait.', flush=True)
    t1 = time()
    with ZstdTarFile(args.output, mode='w', level_or_option=option, zstd_dict=args.zd) as f:
        f.add(args.tar_input_dir, basename)
        uncompressed_size = f.fileobj.tell()
    t2 = time()
    output_file_size = args.output.tell()
    close_files(args)
    if uncompressed_size != 0:
        ratio = 100 * output_file_size / uncompressed_size
    else:
        ratio = 100.0
    msg = 'Archiving succeeded, {:.2f} seconds.\nInput ~{:,} bytes, output {:,} bytes, ratio {:.2f}%.\n'.format(t2 - t1, uncompressed_size, output_file_size, ratio)
    print(msg)