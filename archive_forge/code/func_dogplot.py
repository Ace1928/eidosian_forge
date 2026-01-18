import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def dogplot(*_, **__):
    """Who's a good boy?"""
    from urllib.request import urlopen
    from io import BytesIO
    url = 'https://github.com/mwaskom/seaborn-data/raw/master/png/img{}.png'
    pic = np.random.randint(2, 7)
    data = BytesIO(urlopen(url.format(pic)).read())
    img = plt.imread(data)
    f, ax = plt.subplots(figsize=(5, 5), dpi=100)
    f.subplots_adjust(0, 0, 1, 1)
    ax.imshow(img)
    ax.set_axis_off()