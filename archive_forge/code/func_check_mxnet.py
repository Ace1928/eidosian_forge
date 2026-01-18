import platform, subprocess, sys, os
import socket, time
import argparse
def check_mxnet():
    print('----------MXNet Info-----------')
    try:
        import mxnet
        print('Version      :', mxnet.__version__)
        mx_dir = os.path.dirname(mxnet.__file__)
        print('Directory    :', mx_dir)
        commit_hash = os.path.join(mx_dir, 'COMMIT_HASH')
        if os.path.exists(commit_hash):
            with open(commit_hash, 'r') as f:
                ch = f.read().strip()
                print('Commit Hash   :', ch)
        else:
            print('Commit hash file "{}" not found. Not installed from pre-built package or built from source.'.format(commit_hash))
        print('Library      :', mxnet.libinfo.find_lib_path())
        try:
            print('Build features:')
            print(get_build_features_str())
        except Exception:
            print('No runtime build feature info available')
    except ImportError:
        print('No MXNet installed.')
    except Exception as e:
        import traceback
        if not isinstance(e, IOError):
            print('An error occured trying to import mxnet.')
            print('This is very likely due to missing missing or incompatible library files.')
        print(traceback.format_exc())