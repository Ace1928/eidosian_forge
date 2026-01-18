from rdkit.Chem import ChemicalFeatures
def initFromFeat(self, feat):
    """
    >>> from rdkit import Geometry
    >>> sfeat = ChemicalFeatures.FreeChemicalFeature('Aromatic','Foo',Geometry.Point3D(0,0,0))
    >>> fmp = FeatMapPoint()
    >>> fmp.initFromFeat(sfeat)
    >>> fmp.GetFamily()==sfeat.GetFamily()
    True
    >>> fmp.GetType()==sfeat.GetType()
    True
    >>> list(fmp.GetPos())
    [0.0, 0.0, 0.0]
    >>> fmp.featDirs == []
    True

    >>> sfeat.featDirs = [Geometry.Point3D(1.0,0,0)]
    >>> fmp.initFromFeat(sfeat)
    >>> len(fmp.featDirs)
    1

    """
    self.SetFamily(feat.GetFamily())
    self.SetType(feat.GetType())
    self.SetPos(feat.GetPos())
    if hasattr(feat, 'featDirs'):
        self.featDirs = feat.featDirs[:]