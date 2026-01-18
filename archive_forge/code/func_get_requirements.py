from setuptools import find_packages, setup
def get_requirements(path: str):
    return [l.strip() for l in open(path)]