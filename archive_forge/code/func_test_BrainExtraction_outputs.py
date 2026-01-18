from ..segmentation import BrainExtraction
def test_BrainExtraction_outputs():
    output_map = dict(BrainExtractionBrain=dict(extensions=None), BrainExtractionCSF=dict(extensions=None), BrainExtractionGM=dict(extensions=None), BrainExtractionInitialAffine=dict(extensions=None), BrainExtractionInitialAffineFixed=dict(extensions=None), BrainExtractionInitialAffineMoving=dict(extensions=None), BrainExtractionLaplacian=dict(extensions=None), BrainExtractionMask=dict(extensions=None), BrainExtractionPrior0GenericAffine=dict(extensions=None), BrainExtractionPrior1InverseWarp=dict(extensions=None), BrainExtractionPrior1Warp=dict(extensions=None), BrainExtractionPriorWarped=dict(extensions=None), BrainExtractionSegmentation=dict(extensions=None), BrainExtractionTemplateLaplacian=dict(extensions=None), BrainExtractionTmp=dict(extensions=None), BrainExtractionWM=dict(extensions=None), N4Corrected0=dict(extensions=None), N4Truncated0=dict(extensions=None))
    outputs = BrainExtraction.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value