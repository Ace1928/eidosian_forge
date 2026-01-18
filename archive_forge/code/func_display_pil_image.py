import base64
import copy
import html
import warnings
from io import BytesIO
import IPython
from IPython.display import HTML, SVG
from rdkit import Chem
from rdkit.Chem import Draw, rdchem, rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D
from . import InteractiveRenderer
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from IPython import display
def display_pil_image(img):
    """displayhook function for PIL Images, rendered as PNG"""
    metadata = PngInfo()
    for k, v in img.info.items():
        metadata.add_text(k, v)
    bio = BytesIO()
    img.save(bio, format='PNG', pnginfo=metadata)
    return bio.getvalue()