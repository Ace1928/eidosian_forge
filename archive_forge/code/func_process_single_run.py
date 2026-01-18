import argparse
import json
import os
import re
import shutil
import yaml
def process_single_run(in_dir, out_dir):
    exp_dir = os.listdir(in_dir)
    assert 'params.json' in exp_dir and 'progress.csv' in exp_dir, 'params.json or progress.csv not found in {}!'.format(in_dir)
    os.makedirs(out_dir, exist_ok=True)
    for file in exp_dir:
        absfile = os.path.join(in_dir, file)
        if file == 'params.json':
            assert os.path.isfile(absfile), '{} not a file!'.format(file)
            with open(absfile) as fp:
                contents = json.load(fp)
            with open(os.path.join(out_dir, 'config.yaml'), 'w') as fp:
                yaml.dump(contents, fp)
        elif file == 'progress.csv':
            assert os.path.isfile(absfile), '{} not a file!'.format(file)
            col_idx_to_filter = []
            with open(absfile) as fp:
                col_names_orig = fp.readline().strip().split(',')
                cols_to_filter = args.results_filter.split(',')
                for i, c in enumerate(col_names_orig):
                    if c in cols_to_filter:
                        col_idx_to_filter.insert(0, i)
                col_names = col_names_orig.copy()
                for idx in col_idx_to_filter:
                    col_names.pop(idx)
                absfile_out = os.path.join(out_dir, 'progress.csv')
                with open(absfile_out, 'w') as out_fp:
                    print(','.join(col_names), file=out_fp)
                    while True:
                        line = fp.readline().strip()
                        if not line:
                            break
                        line = re.sub('(,{2,})', lambda m: ',None' * (len(m.group()) - 1) + ',', line)
                        cols = re.findall('".+?"|[^,]+', line)
                        if len(cols) != len(col_names_orig):
                            continue
                        for idx in col_idx_to_filter:
                            cols.pop(idx)
                        print(','.join(cols), file=out_fp)
            out_size = os.path.getsize(absfile_out)
            max_size = args.results_max_size * 1024
            if 0 < max_size < out_size:
                ratio = out_size / max_size
                if ratio > 2.0:
                    nth = out_size // max_size
                    os.system("awk 'NR==1||NR%{}==0' {} > {}.new".format(nth, absfile_out, absfile_out))
                else:
                    nth = out_size // (out_size - max_size)
                    os.system("awk 'NR==1||NR%{}!=0' {} > {}.new".format(nth, absfile_out, absfile_out))
                os.remove(absfile_out)
                os.rename(absfile_out + '.new', absfile_out)
            zip_file = os.path.join(out_dir, 'results.zip')
            try:
                os.remove(zip_file)
            except FileNotFoundError:
                pass
            os.system('zip -j {} {}'.format(zip_file, os.path.join(out_dir, 'progress.csv')))
            os.remove(os.path.join(out_dir, 'progress.csv'))
        elif re.search('^(events\\.out\\.|params\\.pkl)', file):
            assert os.path.isfile(absfile), '{} not a file!'.format(file)
            shutil.copyfile(absfile, os.path.join(out_dir, file))