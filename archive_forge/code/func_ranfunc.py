import os
def ranfunc():
    """A function with some random output.

       Normal examples are verified as usual:
       >>> 1+3
       4

       But if you put '# random' in the output, it is ignored:
       >>> 1+3
       junk goes here...  # random

       >>> 1+2
       again,  anything goes #random
       if multiline, the random mark is only needed once.

       >>> 1+2
       You can also put the random marker at the end:
       # random

       >>> 1+2
       # random
       .. or at the beginning.

       More correct input is properly verified:
       >>> ranfunc()
       'ranfunc'
    """
    return 'ranfunc'