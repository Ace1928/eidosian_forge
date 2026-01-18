import py
import sys
def default_consumer(msg):
    """ the default consumer, prints the message to stdout (using 'print') """
    sys.stderr.write(str(msg) + '\n')