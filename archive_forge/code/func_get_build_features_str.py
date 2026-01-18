import platform, subprocess, sys, os
import socket, time
import argparse
def get_build_features_str():
    import mxnet.runtime
    features = mxnet.runtime.Features()
    return '\n'.join(map(str, list(features.values())))