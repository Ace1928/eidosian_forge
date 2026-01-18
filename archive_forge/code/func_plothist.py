from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
def plothist(x, distfn, args, loc, scale, right=1):
    plt.figure()
    n, bins, patches = plt.hist(x, 25, normed=1, facecolor='green', alpha=0.75)
    maxheight = max([p.get_height() for p in patches])
    print(maxheight)
    axlim = list(plt.axis())
    axlim[-1] = maxheight * 1.05
    yt = distfn.pdf(bins, *args, loc=loc, scale=scale)
    yt[yt > maxheight] = maxheight
    lt = plt.plot(bins, yt, 'r--', linewidth=1)
    ys = stats.t.pdf(bins, 10, scale=10) * right
    ls = plt.plot(bins, ys, 'b-', linewidth=1)
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('$\\mathrm{{Testing: {} :}}\\ \\mu={:f},\\ \\sigma={:f}$'.format(distfn.name, loc, scale))
    plt.grid(True)
    plt.draw()