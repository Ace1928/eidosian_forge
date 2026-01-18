import os
def on_critical(msg):
    print(msg, file=sys.stderr)