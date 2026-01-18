import numpy as np
from scipy.stats.distributions import norm
def generate_linear():
    nclust = 100
    beta = np.array([1, -2, 1], dtype=np.float64)
    r = 0.4
    rx = 0.5
    p = len(beta)
    OUT = open('gee_linear_1.csv', 'w', encoding='utf-8')
    for i in range(nclust):
        n = np.random.randint(3, 6)
        x = np.random.normal(size=(n, p))
        x = rx * np.random.normal() + np.sqrt(1 - rx ** 2) * x
        x[:, 2] = r * x[:, 1] + np.sqrt(1 - r ** 2) * x[:, 2]
        y = np.dot(x, beta) + np.random.normal(size=n)
        for j in range(n):
            OUT.write('%d, %d,' % (i, y[j]))
            OUT.write(','.join(['%.3f' % b for b in x[j, :]]) + '\n')
    OUT.close()