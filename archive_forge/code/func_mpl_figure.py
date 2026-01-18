import asyncio
import contextlib
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from queue import Empty, Queue
from threading import Thread
import numpy as np
import pytest
import requests
from packaging.version import Version
import panel as pn
from panel.io.server import serve
from panel.io.state import state
from panel.pane.alert import Alert
from panel.pane.markup import Markdown
from panel.widgets.button import _ButtonBase
def mpl_figure():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.random.rand(10, 2))
    plt.close(fig)
    return fig