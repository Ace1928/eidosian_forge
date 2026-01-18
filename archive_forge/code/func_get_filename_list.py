from pyparsing import Literal, CaselessLiteral, Word, delimitedList \
def get_filename_list(tf):
    import sys, glob
    if tf == None:
        if len(sys.argv) > 1:
            tf = sys.argv[1:]
        else:
            tf = glob.glob('*.dfm')
    elif type(tf) == str:
        tf = [tf]
    testfiles = []
    for arg in tf:
        testfiles.extend(glob.glob(arg))
    return testfiles