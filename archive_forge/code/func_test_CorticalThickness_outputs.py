from ..segmentation import CorticalThickness
def test_CorticalThickness_outputs():
    output_map = dict(BrainExtractionMask=dict(extensions=None), BrainSegmentation=dict(extensions=None), BrainSegmentationN4=dict(extensions=None), BrainSegmentationPosteriors=dict(), BrainVolumes=dict(extensions=None), CorticalThickness=dict(extensions=None), CorticalThicknessNormedToTemplate=dict(extensions=None), ExtractedBrainN4=dict(extensions=None), SubjectToTemplate0GenericAffine=dict(extensions=None), SubjectToTemplate1Warp=dict(extensions=None), SubjectToTemplateLogJacobian=dict(extensions=None), TemplateToSubject0Warp=dict(extensions=None), TemplateToSubject1GenericAffine=dict(extensions=None))
    outputs = CorticalThickness.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value