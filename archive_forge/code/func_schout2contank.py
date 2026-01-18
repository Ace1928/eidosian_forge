import numpy as np
import matplotlib.pyplot as plt
def schout2contank(a, b, d):
    th = d * b / np.sqrt(a ** 2 - b ** 2)
    k = 1 / (d * np.sqrt(a ** 2 - b ** 2))
    s = np.sqrt(d / np.sqrt(a ** 2 - b ** 2))
    return (th, k, s)