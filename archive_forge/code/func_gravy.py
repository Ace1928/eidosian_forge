import sys
from Bio.SeqUtils import ProtParamData  # Local
from Bio.SeqUtils import IsoelectricPoint  # Local
from Bio.Seq import Seq
from Bio.Data import IUPACData
from Bio.SeqUtils import molecular_weight
def gravy(self, scale='KyteDoolitle'):
    """Calculate the GRAVY (Grand Average of Hydropathy) according to Kyte and Doolitle, 1982.

        Utilizes the given Hydrophobicity scale, by default uses the original
        proposed by Kyte and Doolittle (KyteDoolitle). Other options are:
        Aboderin, AbrahamLeo, Argos, BlackMould, BullBreese, Casari, Cid,
        Cowan3.4, Cowan7.5, Eisenberg, Engelman, Fasman, Fauchere, GoldSack,
        Guy, Jones, Juretic, Kidera, Miyazawa, Parker,Ponnuswamy, Rose,
        Roseman, Sweet, Tanford, Wilson and Zimmerman.

        New scales can be added in ProtParamData.
        """
    selected_scale = ProtParamData.gravy_scales.get(scale, -1)
    if selected_scale == -1:
        raise ValueError(f'scale: {scale} not known')
    total_gravy = sum((selected_scale[aa] for aa in self.sequence))
    return total_gravy / self.length