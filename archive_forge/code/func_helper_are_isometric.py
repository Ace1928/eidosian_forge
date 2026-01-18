import re
import string
def helper_are_isometric(M, N):
    for i in range(100):
        try:
            if M.is_isometric_to(N):
                return
        except:
            pass
        M.randomize()
        N.randomize()
    raise Exception('Could not find isometry')