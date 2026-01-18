import sys
def shellcomplete(context=None, outfile=None):
    if outfile is None:
        outfile = sys.stdout
    if context is None:
        shellcomplete_commands(outfile=outfile)
    else:
        shellcomplete_on_command(context, outfile=outfile)