import greenlet
def g2_run():
    g1.switch()
    print('Falling off end of g2')