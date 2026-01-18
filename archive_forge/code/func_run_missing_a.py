from breezy import osutils, tests
def run_missing_a(args, retcode=1):
    return run_missing(['../a'] + args, retcode=retcode, working_dir='b')