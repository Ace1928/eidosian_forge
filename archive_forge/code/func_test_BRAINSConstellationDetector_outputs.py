from ..specialized import BRAINSConstellationDetector
def test_BRAINSConstellationDetector_outputs():
    output_map = dict(outputLandmarksInACPCAlignedSpace=dict(extensions=None), outputLandmarksInInputSpace=dict(extensions=None), outputMRML=dict(extensions=None), outputResampledVolume=dict(extensions=None), outputTransform=dict(extensions=None), outputUntransformedClippedVolume=dict(extensions=None), outputVerificationScript=dict(extensions=None), outputVolume=dict(extensions=None), resultsDir=dict(), writeBranded2DImage=dict(extensions=None))
    outputs = BRAINSConstellationDetector.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value