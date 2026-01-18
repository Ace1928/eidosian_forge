from breezy import osutils, tests
def run_missing_b(args, retcode=1):
    return run_missing(['../b'] + args, retcode=retcode, working_dir='a')