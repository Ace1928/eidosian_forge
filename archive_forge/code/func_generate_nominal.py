import numpy as np
from scipy.stats.distributions import norm
def generate_nominal():
    beta1 = np.r_[0.5, 0.5]
    beta2 = np.r_[-1, -0.5]
    p = len(beta1)
    rz = 0.5
    OUT = open('gee_nominal_1.csv', 'w', encoding='utf-8')
    for i in range(200):
        n = np.random.randint(3, 6)
        x = np.random.normal(size=(n, p))
        x[:, 0] = 1
        for j in range(1, x.shape[1]):
            x[:, j] += np.random.normal()
        pr1 = np.exp(np.dot(x, beta1))[:, None]
        pr2 = np.exp(np.dot(x, beta2))[:, None]
        den = 1 + pr1 + pr2
        pr = np.hstack((pr1 / den, pr2 / den, 1 / den))
        cpr = np.cumsum(pr, 1)
        z = rz * np.random.normal() + np.sqrt(1 - rz ** 2) * np.random.normal(size=n)
        u = norm.cdf(z)
        y = (u[:, None] > cpr).sum(1)
        for j in range(n):
            OUT.write('%d, %d,' % (i, y[j]))
            OUT.write(','.join(['%.3f' % b for b in x[j, :]]) + '\n')
    OUT.close()