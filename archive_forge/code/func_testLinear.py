import numpy
def testLinear():
    examples = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.8, 0.8]]
    net = Network.Network([1, 2, 1])
    t = BackProp(speed=0.8)
    t.TrainOnLine(examples, net, errTol=0.1, useAvgErr=0)
    print('classifications:')
    for example in examples:
        res = net.ClassifyExample(example[:-1])
        print('%f -> %f' % (example[-1], res))
    return net